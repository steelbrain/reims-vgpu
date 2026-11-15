//! Copying a directory of one volume into a new volume, record for record: the directory becomes
//! the new volume's root, and every object below it keeps its mode, owner, flags, times,
//! attributes and compressed form, renumbered into the new volume, its data streams' bytes read
//! from where the source keeps them.
//!
//! A caller may replace any entry with a symbolic link (nothing below a replaced directory is
//! copied), and add files of its own in directories the copy holds. This is how a restore makes
//! the Data volume from the system image's data template (docs/design.md, "The disk").
//!
//! The records copied are those a data template holds: inodes (with their name and data-stream
//! fields), directory entries, attributes (held in their record, or in a data stream such as a
//! resource fork's), data-stream identifiers and file extents. A hard link, a sibling record or a
//! directory's statistics fails the copy rather than being copied wrong.

use std::collections::HashMap;

use crate::fs::{self, key_header, key_word};
use crate::ondisk::{ino, rec};

use super::fsbuild::{self, SYMLINK_XATTR};
use super::records::{DT_LNK, INODE_FLAGS_BASE, InodeBuilder, dir_rec, name_xfield, xfields};
use super::volume::{Counts, Stream};

/// What becomes of an entry the copy reaches.
pub enum Replace {
    /// Copied as it is (a directory with what is below it).
    Keep,
    /// Replaced by a symbolic link to this target; nothing below it is copied.
    Link(Vec<u8>),
}

/// A run of a source stream: `len` bytes from source block `phys`, or a hole.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceRun {
    pub len: u64,
    pub phys: Option<u64>,
}

/// A copied tree: its records (but file extents), its data streams, where each stream's bytes are
/// in the source, in order, what the volume counts, and the next identifier.
pub struct Copied {
    pub records: Vec<(Vec<u8>, Vec<u8>)>,
    pub streams: Vec<Stream>,
    /// For each stream, its runs in the source; or, for a file the caller added, the index of
    /// that file among the additions.
    pub sources: Vec<StreamSource>,
    pub counts: Counts,
    pub next_obj_id: u64,
}

/// Where a copied stream's bytes come from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StreamSource {
    /// The source volume's blocks.
    Runs(Vec<SourceRun>),
    /// The caller's added file of this index.
    Added(usize),
}

/// A tree the copy cannot make.
#[derive(Debug, thiserror::Error)]
pub enum CopyError {
    #[error("object {0:#x}: the copy has no {1} record kind")]
    Unsupported(u64, &'static str),
    #[error("object {0:#x} has no inode")]
    NoInode(u64),
    #[error("{0}: added in a directory the copy does not hold")]
    NoParent(String),
    #[error("stream {0:#x}: its extents do not cover it from its start")]
    Gap(u64),
}

/// A record: its key and its value.
type Record = (Vec<u8>, Vec<u8>);

/// The records of each object, by identifier.
struct Index<'a> {
    by_oid: HashMap<u64, Vec<&'a Record>>,
}

impl<'a> Index<'a> {
    fn new(records: &'a [(Vec<u8>, Vec<u8>)]) -> Self {
        let mut by_oid: HashMap<u64, Vec<&Record>> = HashMap::new();
        for r in records {
            by_oid.entry(key_header(&r.0).0).or_default().push(r);
        }
        Self { by_oid }
    }

    fn of(&self, oid: u64) -> &[&'a (Vec<u8>, Vec<u8>)] {
        self.by_oid.get(&oid).map_or(&[], Vec::as_slice)
    }
}

fn u64_at(b: &[u8], at: usize) -> u64 {
    u64::from_le_bytes(b[at..at + 8].try_into().unwrap())
}

fn put_u64(b: &mut [u8], at: usize, v: u64) {
    b[at..at + 8].copy_from_slice(&v.to_le_bytes());
}

/// A record's key with its identifier replaced, the rest kept.
fn rekey(k: &[u8], oid: u64) -> Vec<u8> {
    let (_, t) = key_header(k);
    let mut out = k.to_vec();
    out[..8].copy_from_slice(&key_word(oid, t).to_le_bytes());
    out
}

/// The runs of stream `id` in the source, from its file extents, in order.
fn runs_of(idx: &Index, id: u64) -> Result<Vec<SourceRun>, CopyError> {
    let mut ext: Vec<(u64, u64, u64)> = idx
        .of(id)
        .iter()
        .filter(|(k, _)| key_header(k).1 == rec::FILE_EXTENT)
        .map(|(k, v)| {
            let e = fs::FileExtent::parse(k, v);
            (e.logical, e.len, e.phys)
        })
        .collect();
    ext.sort_unstable();
    let mut at = 0;
    let mut runs = Vec::with_capacity(ext.len());
    for (logical, len, phys) in ext {
        if logical != at {
            return Err(CopyError::Gap(id));
        }
        runs.push(SourceRun {
            len,
            phys: (phys != 0).then_some(phys),
        });
        at += len;
    }
    Ok(runs)
}

/// The total length of runs.
fn runs_len(runs: &[SourceRun]) -> u64 {
    runs.iter().map(|r| r.len).sum()
}

/// Copy the tree below directory `root` of the volume whose records are `records` (all of them,
/// in any order) into a new volume, names hashed folding case where `fold`. `decide` is asked,
/// for each entry below `root` that is not itself a symbolic link, by its path from `root` and
/// whether it is a directory, whether to copy it or make it a link; a link made is stamped `t`.
/// `added` are files to add, each in a directory the copy holds.
///
/// # Errors
///
/// [`CopyError`] for a record kind the copy does not make, an object without its inode, a
/// stream whose extents leave a gap, or an addition whose directory the copy lacks.
pub fn copy_tree(
    records: &[(Vec<u8>, Vec<u8>)],
    root: u64,
    fold: bool,
    t: u64,
    decide: &mut dyn FnMut(&str, bool) -> Replace,
    added: &[fsbuild::Item],
) -> Result<Copied, CopyError> {
    let idx = Index::new(records);
    let mut out = Copied {
        records: Vec::new(),
        streams: Vec::new(),
        sources: Vec::new(),
        counts: Counts::default(),
        next_obj_id: ino::MIN_USER,
    };
    let new_oid = |out: &mut Copied| {
        let o = out.next_obj_id;
        out.next_obj_id += 1;
        o
    };
    // Each copied directory's path and new identifier, for the additions.
    let mut dirs: HashMap<String, u64> = HashMap::from([(String::new(), ino::ROOT_DIR)]);
    let mut children: HashMap<u64, i32> = HashMap::new();

    // Depth first: (source object, new identifier, new parent, path).
    let mut stack: Vec<(u64, u64, u64, String)> = vec![(root, ino::ROOT_DIR, 1, String::new())];
    while let Some((src, oid, parent, path)) = stack.pop() {
        let recs = idx.of(src);
        let inode = recs
            .iter()
            .find(|(k, _)| key_header(k).1 == rec::INODE)
            .ok_or(CopyError::NoInode(src))?;
        let parsed = fs::Inode::parse(&inode.1).ok_or(CopyError::NoInode(src))?;
        if parsed.private_id != src {
            return Err(CopyError::Unsupported(src, "shared data stream"));
        }
        if !parsed.is_dir() && parsed.nchildren_or_nlink > 1 {
            return Err(CopyError::Unsupported(src, "hard link"));
        }
        let is_dir = parsed.is_dir();
        match parsed.mode & 0o170000 {
            0o040000 => out.counts.directories += u64::from(oid != ino::ROOT_DIR),
            0o100000 => out.counts.files += 1,
            0o120000 => out.counts.symlinks += 1,
            _ => out.counts.other += 1,
        }
        if is_dir {
            dirs.insert(path.clone(), oid);
        }
        for (k, v) in recs {
            match key_header(k).1 {
                rec::INODE => {
                    let mut v = v.clone();
                    put_u64(&mut v, 0, parent);
                    put_u64(&mut v, 8, oid);
                    if oid == ino::ROOT_DIR {
                        // The volume's root names itself `root`.
                        v.truncate(fs::INODE_FIXED);
                        let mut fields: Vec<_> = parsed
                            .xfields
                            .iter()
                            .filter(|x| x.kind != fs::xf::NAME)
                            .map(|x| super::records::XField {
                                kind: x.kind,
                                flags: x.flags,
                                data: x.data.clone(),
                            })
                            .collect();
                        fields.insert(0, name_xfield(b"root"));
                        v.extend_from_slice(&xfields(&fields));
                    }
                    out.records.push((rekey(k, oid), v));
                }
                rec::DIR_REC => {
                    let child = u64_at(v, 0);
                    let name = fs::drec_name(k);
                    let cpath = if path.is_empty() {
                        String::from_utf8_lossy(name).into_owned()
                    } else {
                        format!("{path}/{}", String::from_utf8_lossy(name))
                    };
                    let child_inode = idx
                        .of(child)
                        .iter()
                        .find(|(k, _)| key_header(k).1 == rec::INODE)
                        .and_then(|(_, v)| fs::Inode::parse(v))
                        .ok_or(CopyError::NoInode(child))?;
                    let child_is_dir = child_inode.is_dir();
                    let is_link = child_inode.mode & 0o170000 == 0o120000;
                    let replace = if is_link {
                        Replace::Keep
                    } else {
                        decide(&cpath, child_is_dir)
                    };
                    let new_child = new_oid(&mut out);
                    *children.entry(oid).or_default() += 1;
                    let mut val = v.clone();
                    put_u64(&mut val, 0, new_child);
                    match replace {
                        Replace::Keep => {
                            out.records.push((rekey(k, oid), val));
                            stack.push((child, new_child, oid, cpath));
                        }
                        Replace::Link(target) => {
                            out.records
                                .push(dir_rec(oid, name, fold, new_child, t, DT_LNK));
                            out.records.extend(link_records(
                                new_child,
                                oid,
                                name,
                                &target,
                                t,
                                &child_inode,
                            ));
                            out.counts.symlinks += 1;
                        }
                    }
                }
                rec::XATTR => {
                    let mut val = v.clone();
                    if let Some((_, fs::XattrData::Stream(sid, _))) = fs::parse_xattr(v) {
                        let runs = runs_of(&idx, sid)?;
                        let nsid = new_oid(&mut out);
                        put_u64(&mut val, 4, nsid);
                        out.streams.push(Stream {
                            id: nsid,
                            owner: oid,
                            len: runs_len(&runs),
                        });
                        out.sources.push(StreamSource::Runs(runs));
                    }
                    out.records.push((rekey(k, oid), val));
                }
                rec::DSTREAM_ID => {
                    let runs = runs_of(&idx, src)?;
                    out.streams.push(Stream {
                        id: oid,
                        owner: oid,
                        len: runs_len(&runs),
                    });
                    out.sources.push(StreamSource::Runs(runs));
                    out.records.push((rekey(k, oid), v.clone()));
                }
                rec::FILE_EXTENT => {}
                rec::DIR_STATS => return Err(CopyError::Unsupported(src, "directory statistics")),
                rec::SIBLING_LINK | rec::SIBLING_MAP => {
                    return Err(CopyError::Unsupported(src, "sibling"));
                }
                other => return Err(CopyError::Unsupported(src, rec::name(other))),
            }
        }
    }

    // The additions, each a new entry of its directory.
    for (i, it) in added.iter().enumerate() {
        let (dir, name) = it.path.rsplit_once('/').unwrap_or(("", &it.path));
        let parent = *dirs
            .get(dir)
            .ok_or_else(|| CopyError::NoParent(it.path.clone()))?;
        let oid = new_oid(&mut out);
        *children.entry(parent).or_default() += 1;
        let len = match it.kind {
            fsbuild::Kind::File { len } => len,
            _ => {
                return Err(CopyError::Unsupported(
                    oid,
                    "added entry that is not a file",
                ));
            }
        };
        let mut fields = vec![name_xfield(name.as_bytes())];
        if len > 0 {
            fields.push(super::records::dstream_xfield(super::records::dstream(
                len,
                super::volume::blocks_of(len) * super::geometry::BLOCK_SIZE,
                0,
                len,
                0,
            )));
            out.streams.push(Stream {
                id: oid,
                owner: oid,
                len,
            });
            out.sources.push(StreamSource::Added(i));
            out.records.push((
                key_word(oid, rec::DSTREAM_ID).to_le_bytes().to_vec(),
                1u32.to_le_bytes().to_vec(),
            ));
        }
        out.records.push(dir_rec(
            parent,
            name.as_bytes(),
            fold,
            oid,
            t,
            super::records::DT_REG,
        ));
        out.records.push(
            InodeBuilder {
                oid,
                parent,
                private_id: oid,
                create_time: t,
                mod_time: t,
                change_time: t,
                access_time: t,
                internal_flags: INODE_FLAGS_BASE,
                nchildren_or_nlink: 1,
                default_protection_class: 0,
                write_generation_counter: 1,
                bsd_flags: it.bsd_flags,
                owner: it.uid,
                group: it.gid,
                mode: 0o100000 | (it.mode & 0o7777),
                uncompressed_size: 0,
                xfields: fields,
            }
            .build(),
        );
        out.counts.files += 1;
    }

    // A directory that gained an addition counts it.
    for (k, v) in &mut out.records {
        let (oid, t) = key_header(k);
        if t == rec::INODE
            && let (Some(n), Some(inode)) = (children.get(&oid), fs::Inode::parse(v))
            && inode.is_dir()
        {
            v[56..60].copy_from_slice(&n.to_le_bytes());
        }
    }
    // The volume's private directory and its entry, as every volume has.
    for (k, v) in super::records::empty_volume_records(fold, t) {
        let (oid, kind) = key_header(&k);
        let is_root = (kind == rec::INODE && oid == ino::ROOT_DIR)
            || (kind == rec::DIR_REC && fs::drec_name(&k) == b"root");
        if !is_root || kind == rec::DIR_REC {
            out.records.push((k, v));
        }
    }
    out.records
        .sort_by(|a, b| crate::fs::compare_keys(&a.0, &b.0));
    Ok(out)
}

/// A symbolic link's inode and its target attribute, in place of `was`: the link keeps its
/// permissions, owner and flags (but compression, which a link's content never has), as VZ's
/// restore makes its links into the data template **[vz]**.
fn link_records(
    oid: u64,
    parent: u64,
    name: &[u8],
    target: &[u8],
    t: u64,
    was: &fs::Inode,
) -> Vec<(Vec<u8>, Vec<u8>)> {
    let mut v = target.to_vec();
    v.push(0);
    let mut xk = key_word(oid, rec::XATTR).to_le_bytes().to_vec();
    xk.extend_from_slice(&((SYMLINK_XATTR.len() + 1) as u16).to_le_bytes());
    xk.extend_from_slice(SYMLINK_XATTR);
    xk.push(0);
    let mut xv = Vec::with_capacity(4 + v.len());
    xv.extend_from_slice(&(fs::XATTR_DATA_EMBEDDED | fs::XATTR_FILE_SYSTEM_OWNED).to_le_bytes());
    xv.extend_from_slice(&(v.len() as u16).to_le_bytes());
    xv.extend_from_slice(&v);
    vec![
        InodeBuilder {
            oid,
            parent,
            private_id: oid,
            create_time: t,
            mod_time: t,
            change_time: t,
            access_time: t,
            internal_flags: INODE_FLAGS_BASE,
            nchildren_or_nlink: 1,
            default_protection_class: 0,
            write_generation_counter: 1,
            bsd_flags: was.bsd_flags & !crate::decmpfs::UF_COMPRESSED,
            owner: was.owner,
            group: was.group,
            mode: 0o120000 | (was.mode & 0o7777),
            uncompressed_size: 0,
            xfields: vec![name_xfield(name)],
        }
        .build(),
        (xk, xv),
    ]
}
