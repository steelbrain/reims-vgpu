//! Files by path: opening a disk or container, resolving a path in a volume, and reading a
//! file's content, decoded where it is compressed.

use std::sync::Arc;

use crate::dev::{Partition, Source};
use crate::fs::{self, DirEntry, Inode, XattrData};
use crate::le::u64_at;
use crate::ondisk::{NX_MAGIC, apfs, ino, part, rec};
use crate::read::{Container, Volume};
use crate::{Error, Result, corrupt, decmpfs, gpt, names};

/// The attribute a symbolic link's target is held in.
const SYMLINK_XATTR: &[u8] = b"com.apple.fs.symlink";
/// The most symbolic links one path resolution follows.
const MAX_SYMLINKS: usize = 32;

const S_IFMT: u16 = 0o170_000;
const S_IFLNK: u16 = 0o120_000;
const S_IFREG: u16 = 0o100_000;

/// Open the APFS container in `src`: `src` itself where it starts with a container
/// superblock, or else the first APFS partition of its GUID partition table.
///
/// # Errors
///
/// `src` is neither a container nor a disk with an APFS partition, or the container fails its
/// checks.
pub fn open(src: Arc<dyn Source>) -> Result<Container> {
    let mut head = [0u8; 36];
    src.read_at(0, &mut head)?;
    if u32::from_le_bytes(head[32..36].try_into().unwrap()) == NX_MAGIC {
        return Container::open(Partition::whole(src));
    }
    let table = gpt::read(src.as_ref())
        .map_err(|e| Error::Invalid(format!("neither an APFS container nor a GPT disk ({e})")))?;
    let p = table
        .partitions
        .iter()
        .find(|p| p.type_guid == part::APFS)
        .ok_or_else(|| Error::Invalid("the disk's partition table has no APFS partition".into()))?;
    let (base, len) = p.byte_range(table.sector_size);
    if base.checked_add(len).is_none_or(|end| end > src.len()) {
        return Err(Error::Gpt(format!(
            "partition {} runs past the disk",
            p.index
        )));
    }
    Container::open(Partition::new(src, base, len))
}

/// A file-system object found by path: its inode number and inode.
#[derive(Clone, Debug)]
pub struct Entry {
    pub id: u64,
    pub inode: Inode,
}

impl Entry {
    #[must_use]
    pub const fn is_dir(&self) -> bool {
        self.inode.is_dir()
    }

    #[must_use]
    pub const fn is_file(&self) -> bool {
        self.inode.mode & S_IFMT == S_IFREG
    }

    #[must_use]
    pub const fn is_symlink(&self) -> bool {
        self.inode.mode & S_IFMT == S_IFLNK
    }

    /// Whether the file's content is held compressed (`UF_COMPRESSED`).
    #[must_use]
    pub const fn is_compressed(&self) -> bool {
        self.inode.bsd_flags & decmpfs::UF_COMPRESSED != 0
    }
}

/// A run of a data stream's bytes on the container.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Extent {
    /// Where in the stream the run starts.
    pub logical: u64,
    /// The run's length in bytes.
    pub len: u64,
    /// Where the run starts, in bytes from the start of the source the container was opened
    /// on; `None` for a hole, which reads as zeros.
    pub offset: Option<u64>,
}

impl Volume<'_> {
    /// The object at `path`, following symbolic links. `path` is from the volume's root; `.`
    /// and `..` are taken as they come, and names compare as the volume compares them.
    ///
    /// # Errors
    ///
    /// A component is missing or is not a directory, links loop, or a record fails its
    /// checks.
    pub fn lookup(&self, path: &str) -> Result<Entry> {
        self.check_names()?;
        let root = self.inode(ino::ROOT_DIR)?;
        let mut dirs = vec![root.clone()];
        let mut cur = root;
        let mut todo: Vec<String> = path
            .split('/')
            .rev()
            .filter(|c| !c.is_empty())
            .map(str::to_owned)
            .collect();
        let mut links = 0;
        while let Some(name) = todo.pop() {
            if !cur.is_dir() {
                return Err(Error::Invalid(format!(
                    "{path}: a component before {name:?} is not a directory"
                )));
            }
            match name.as_str() {
                "." => continue,
                ".." => {
                    if dirs.len() > 1 {
                        dirs.pop();
                    }
                    cur = dirs.last().unwrap().clone();
                    continue;
                }
                _ => {}
            }
            let id = self
                .dir_lookup(cur.id, &name)?
                .ok_or_else(|| Error::Invalid(format!("{path}: no {name:?} in its directory")))?;
            let e = self.inode(id)?;
            if e.is_symlink() {
                links += 1;
                if links > MAX_SYMLINKS {
                    return Err(Error::Invalid(format!(
                        "{path}: more than {MAX_SYMLINKS} symbolic links"
                    )));
                }
                let target = self
                    .xattr(&e, SYMLINK_XATTR)?
                    .ok_or_else(|| corrupt(0, format!("symbolic link {id} without its target")))?;
                let target = target.strip_suffix(&[0]).unwrap_or(&target);
                let target = std::str::from_utf8(target).map_err(|_| {
                    Error::Invalid(format!("{path}: a link target that is not UTF-8"))
                })?;
                if target.starts_with('/') {
                    dirs.truncate(1);
                    cur = dirs[0].clone();
                }
                todo.extend(
                    target
                        .split('/')
                        .rev()
                        .filter(|c| !c.is_empty())
                        .map(str::to_owned),
                );
                continue;
            }
            if e.is_dir() {
                dirs.push(e.clone());
            }
            cur = e;
        }
        Ok(cur)
    }

    /// The content of the file at `path`, following symbolic links, decoded where it is
    /// compressed.
    ///
    /// # Errors
    ///
    /// The path does not resolve to a regular file, its content is compressed in a way this
    /// crate does not decode, or a record or block fails its checks.
    pub fn read_file(&self, path: &str) -> Result<Vec<u8>> {
        let e = self.lookup(path)?;
        if !e.is_file() {
            return Err(Error::Invalid(format!(
                "{path} is not a regular file (mode {:o})",
                e.inode.mode
            )));
        }
        self.read(&e)
    }

    /// The content of regular file `e`, decoded where it is compressed.
    ///
    /// # Errors
    ///
    /// As [`read_file`](Self::read_file).
    pub fn read(&self, e: &Entry) -> Result<Vec<u8>> {
        if !e.is_compressed() {
            let size = e.inode.dstream().map_or(0, |d| d.size);
            return self.read_stream(e.inode.private_id, size);
        }
        let attr = self
            .xattr(e, decmpfs::XATTR)?
            .ok_or_else(|| corrupt(0, format!("compressed file {} without its header", e.id)))?;
        let h = decmpfs::Header::parse(&attr)
            .ok_or_else(|| corrupt(0, format!("compressed file {}: no fpmc header", e.id)))?;
        let fork = if h.in_resource_fork() {
            Some(self.xattr(e, decmpfs::RESOURCE_FORK)?.ok_or_else(|| {
                corrupt(
                    0,
                    format!("compressed file {} without its resource fork", e.id),
                )
            })?)
        } else {
            None
        };
        decmpfs::decode(&attr, fork.as_deref())
    }

    /// The runs of `e`'s data stream on the container, in order; for an uncompressed file,
    /// where its content is. Runs past the stream's size are cut to it.
    ///
    /// # Errors
    ///
    /// The extent records fail their checks.
    pub fn extents(&self, e: &Entry) -> Result<Vec<Extent>> {
        let size = e.inode.dstream().map_or(0, |d| d.size);
        let (base, bs) = (self.c.base(), u64::from(self.c.block_size()));
        Ok(self
            .stream_extents(e.inode.private_id, size)?
            .into_iter()
            .map(|r| Extent {
                logical: r.logical,
                len: r.len,
                offset: (r.phys != 0).then(|| base + r.phys * bs),
            })
            .collect())
    }

    /// Attribute `name` of `e`, read whole: held in its record, or in a data stream.
    ///
    /// # Errors
    ///
    /// The record or its stream fail their checks.
    pub fn xattr(&self, e: &Entry, name: &[u8]) -> Result<Option<Vec<u8>>> {
        let mut start = vec![0u8; 10];
        start[..8].copy_from_slice(&fs::key_word(e.id, rec::XATTR).to_le_bytes());
        let mut found = None;
        self.fs_tree()?.scan(&start, &mut |k, v| {
            if fs::key_header(k) != (e.id, rec::XATTR) {
                return Ok(false);
            }
            if fs::xattr_name(k) == name {
                found = Some(
                    fs::parse_xattr(v)
                        .ok_or_else(|| {
                            corrupt(0, format!("attribute of {} with a malformed value", e.id))
                        })?
                        .1,
                );
                return Ok(false);
            }
            Ok(true)
        })?;
        match found {
            None => Ok(None),
            Some(XattrData::Embedded(d)) => Ok(Some(d)),
            Some(XattrData::Stream(id, d)) => self.read_stream(id, d.size).map(Some),
        }
    }

    fn check_names(&self) -> Result<()> {
        if self.sb.incompat
            & (apfs::INCOMPAT_CASE_INSENSITIVE | apfs::INCOMPAT_NORMALIZATION_INSENSITIVE)
            == 0
        {
            return Err(Error::Unsupported(
                "a volume whose directory entries are not keyed by a hash of their names".into(),
            ));
        }
        if self.sb.fs_flags & apfs::FS_UNENCRYPTED == 0 {
            return Err(Error::Unsupported("an encrypted volume".into()));
        }
        Ok(())
    }

    /// The inode numbered `id`.
    fn inode(&self, id: u64) -> Result<Entry> {
        let start = fs::key_word(id, rec::INODE).to_le_bytes();
        let mut found = None;
        self.fs_tree()?.scan(&start, &mut |k, v| {
            if fs::key_header(k) == (id, rec::INODE) {
                found = Some(
                    Inode::parse(v)
                        .ok_or_else(|| corrupt(0, format!("inode {id} with a short value")))?,
                );
            }
            Ok(false)
        })?;
        let inode = found.ok_or_else(|| corrupt(0, format!("no inode {id}")))?;
        Ok(Entry { id, inode })
    }

    /// The inode number directory `dir`'s entry `name` names.
    fn dir_lookup(&self, dir: u64, name: &str) -> Result<Option<u64>> {
        let fold = self.sb.case_insensitive();
        let hash = names::hash(name.as_bytes(), fold);
        let mut start = vec![0u8; 12];
        start[..8].copy_from_slice(&fs::key_word(dir, rec::DIR_REC).to_le_bytes());
        start[8..12].copy_from_slice(&(hash << 10).to_le_bytes());
        let mut found = None;
        self.fs_tree()?.scan(&start, &mut |k, v| {
            if k.len() < 12 || fs::key_header(k) != (dir, rec::DIR_REC) || fs::drec_hash(k) != hash
            {
                return Ok(false);
            }
            if names::same(fs::drec_name(k), name.as_bytes(), fold) {
                found = Some(
                    DirEntry::parse(v)
                        .ok_or_else(|| {
                            corrupt(0, format!("directory entry in {dir} with a short value"))
                        })?
                        .file_id,
                );
                return Ok(false);
            }
            Ok(true)
        })?;
        Ok(found)
    }

    /// The runs of data stream `id` of `size` bytes: from a sealed volume's file extent tree, or
    /// else from the file-system tree's extent records.
    fn stream_extents(&self, id: u64, size: u64) -> Result<Vec<Run>> {
        let blocks = self.c.superblock().block_count;
        let bs = u64::from(self.c.block_size());
        let mut out: Vec<Run> = Vec::new();
        let mut push = |logical: u64, len: u64, phys: u64| -> Result<bool> {
            let end = out.last().map_or(0, |x| x.logical + x.len);
            if logical < end {
                return Err(corrupt(
                    phys,
                    format!("extent of stream {id} at {logical} overlaps the one before it"),
                ));
            }
            if logical >= size {
                return Ok(false);
            }
            let len = len.min(size - logical);
            if phys != 0
                && phys
                    .checked_add(len.div_ceil(bs))
                    .is_none_or(|e| e > blocks)
            {
                return Err(corrupt(
                    phys,
                    format!("extent of stream {id} runs past the container"),
                ));
            }
            out.push(Run { logical, len, phys });
            Ok(true)
        };
        if let Some(t) = self.fext_tree()? {
            let mut start = [0u8; 16];
            start[..8].copy_from_slice(&id.to_le_bytes());
            t.scan(&start, &mut |k, v| {
                if u64_at(k, 0) != id {
                    return Ok(false);
                }
                if k.len() < 16 || v.len() < 16 {
                    return Err(corrupt(0, "short file extent tree record"));
                }
                let r = fs::FextRecord::parse(k, v);
                push(r.logical, r.len, r.phys)
            })?;
        } else {
            let mut start = [0u8; 16];
            start[..8].copy_from_slice(&fs::key_word(id, rec::FILE_EXTENT).to_le_bytes());
            self.fs_tree()?.scan(&start, &mut |k, v| {
                if fs::key_header(k) != (id, rec::FILE_EXTENT) {
                    return Ok(false);
                }
                if k.len() < 16 || v.len() < 24 {
                    return Err(corrupt(0, "short file extent record"));
                }
                let r = fs::FileExtent::parse(k, v);
                push(r.logical, r.len, r.phys)
            })?;
        }
        Ok(out)
    }

    /// Data stream `id`'s `size` bytes; holes and anything past the last extent read as zeros.
    fn read_stream(&self, id: u64, size: u64) -> Result<Vec<u8>> {
        let extents = self.stream_extents(id, size)?;
        let n = usize::try_from(size)
            .map_err(|_| Error::Unsupported(format!("a stream of {size} bytes")))?;
        let mut out = Vec::new();
        out.try_reserve_exact(n)
            .map_err(|_| Error::Unsupported(format!("a stream of {size} bytes")))?;
        out.resize(n, 0);
        for x in extents {
            if x.phys != 0 {
                let at = x.logical as usize;
                self.c
                    .read_bytes(x.phys, 0, &mut out[at..at + x.len as usize])?;
            }
        }
        Ok(out)
    }
}

/// A run of a data stream: where in the stream, how many bytes, and its first block (0 for a
/// hole).
struct Run {
    logical: u64,
    len: u64,
    phys: u64,
}
