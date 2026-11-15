//! Canonicalising an unsealed system volume's file-system records into the form the seal covers:
//! every timestamp set to the mtree time `T`, write generations and stream byte counters fixed,
//! file extents taken out of the tree (they move to the physical file-extent tree), and the
//! per-segment content hashes (`file_info`) added. Measured against Apple's sealed system volumes
//! of macOS 14 to 27: every inode time, every directory-entry `date_added`, and the data-stream
//! and attribute-stream byte counters become constants, and the tree holds no file extents.

use crate::fs::{self, INODE_FIXED, xf};
use crate::le::{u16_at, u64_at};
use crate::ondisk::rec;
use crate::seal::Record;

/// The data-stream and attribute-stream byte counters a sealed volume records: both one.
const STREAM_COUNTER: u64 = 1;
/// A sealed inode's write generation.
const WRITE_GEN: u32 = 1;

/// Set the little-endian `u64` at `off` in `v`, if it fits.
fn put_u64(v: &mut [u8], off: usize, x: u64) {
    if off + 8 <= v.len() {
        v[off..off + 8].copy_from_slice(&x.to_le_bytes());
    }
}

fn put_u32(v: &mut [u8], off: usize, x: u32) {
    if off + 4 <= v.len() {
        v[off..off + 4].copy_from_slice(&x.to_le_bytes());
    }
}

/// Canonicalise an inode value in place to time `t` (nanoseconds): its four timestamps, its
/// write generation, and the byte counters of its data-stream extended field.
fn canon_inode(v: &mut [u8], t: u64) {
    if v.len() < INODE_FIXED {
        return;
    }
    // create, modify, change, access times.
    for off in [16, 24, 32, 40] {
        put_u64(v, off, t);
    }
    // write_generation_counter.
    put_u32(v, 64, WRITE_GEN);
    // The data-stream extended field (xf kind DSTREAM) carries total_bytes_written at +24 and
    // total_bytes_read at +32 within its 40-byte j_dstream_t; set both to one. Walk the xfields
    // to find the field's byte range in the raw value.
    let xf_blob = &v[INODE_FIXED..];
    if xf_blob.len() < 4 {
        return;
    }
    let n = usize::from(u16_at(xf_blob, 0));
    let mut data = 4 + n * 4;
    let mut found: Option<usize> = None;
    for i in 0..n {
        let h = 4 + i * 4;
        if h + 4 > xf_blob.len() {
            break;
        }
        let kind = xf_blob[h];
        let size = usize::from(u16_at(xf_blob, h + 2));
        if data + size > xf_blob.len() {
            break;
        }
        if kind == xf::DSTREAM {
            found = Some(INODE_FIXED + data);
        }
        data += size.div_ceil(8) * 8;
    }
    if let Some(ds) = found {
        put_u64(v, ds + 24, STREAM_COUNTER);
        put_u64(v, ds + 32, STREAM_COUNTER);
    }
}

/// Canonicalise a directory-entry value in place: its `date_added` becomes `t`.
fn canon_dir_rec(v: &mut [u8], t: u64) {
    put_u64(v, 8, t);
}

/// Canonicalise an attribute value in place: a stream attribute's data-stream byte counters
/// become one. The value is flags(2) + len(2) + data; a stream attribute's data is the stream
/// identifier (8) then a j_dstream_t whose total_bytes_written is at +24 and total_bytes_read at
/// +32, i.e. offsets 36 and 44 of the value.
fn canon_xattr(v: &mut [u8], block_size: u64) {
    if v.len() < 4 {
        return;
    }
    let flags = u16_at(v, 0);
    if flags & fs::XATTR_DATA_STREAM == 0 {
        return;
    }
    // j_dstream_t within the value: size at +12, allocated_size at +20. Sealing tightens the
    // allocated size to the size rounded up to a block (an unsealed stream may over-allocate).
    let size = u64_at(v, 12);
    let alloced = size.div_ceil(block_size) * block_size;
    put_u64(v, 20, alloced);
    put_u64(v, 36, STREAM_COUNTER);
    put_u64(v, 44, STREAM_COUNTER);
}

/// Canonicalise a key-ordered record stream: patch inode, directory-entry and stream-attribute
/// values to the sealed constants, and drop file-extent records (they move to the file-extent
/// tree). `t` is the mtree time in nanoseconds. The `file_info` content hashes are added
/// separately (they need the files' decoded content). The result stays in key order.
#[must_use]
pub fn canonicalize(records: &[Record], t: u64, block_size: u64) -> Vec<Record> {
    let mut out = Vec::with_capacity(records.len());
    for r in records {
        let ty = fs::key_header(&r.key).1;
        if ty == rec::FILE_EXTENT {
            continue;
        }
        let mut val = r.val.clone();
        match ty {
            rec::INODE => canon_inode(&mut val, t),
            rec::DIR_REC => canon_dir_rec(&mut val, t),
            rec::XATTR => canon_xattr(&mut val, block_size),
            _ => {}
        }
        out.push(Record {
            key: r.key.clone(),
            val,
        });
    }
    out
}

/// A sealed volume hashes each data stream's content in 16 KiB segments.
pub const SEGMENT: usize = 16 * 1024;

/// The `file_info` records for a data stream of `content` bytes, keyed by `stream_id`: one per
/// 16 KiB segment, its value the SHA-256 of the segment zero-padded to whole `block_size` blocks,
/// after a 16-bit block count and the hash length (Apple File System Reference, "Sealed Volumes";
/// the segment and padding are measured against Apple's sealed system volumes).
#[must_use]
pub fn file_infos(stream_id: u64, content: &[u8], block_size: usize) -> Vec<Record> {
    use sha2::{Digest, Sha256};
    let mut out = Vec::new();
    let mut off = 0usize;
    while off < content.len() {
        let end = (off + SEGMENT).min(content.len());
        let seg = &content[off..end];
        let blocks = seg.len().div_ceil(block_size);
        let mut buf = vec![0u8; blocks * block_size];
        buf[..seg.len()].copy_from_slice(seg);
        let hash: [u8; 32] = Sha256::digest(&buf).into();
        let mut key = Vec::with_capacity(16);
        key.extend_from_slice(&fs::key_word(stream_id, rec::FILE_INFO).to_le_bytes());
        key.extend_from_slice(&((1u64 << 56) | off as u64).to_le_bytes());
        let mut val = Vec::with_capacity(35);
        val.extend_from_slice(&(blocks as u16).to_le_bytes());
        val.push(32);
        val.extend_from_slice(&hash);
        out.push(Record { key, val });
        off = end;
    }
    out
}

/// The content stream a file's `file_info` hashes, and its stream identifier: an uncompressed
/// file's data stream (keyed by the inode id), or a resource-fork-compressed file's decoded
/// content (keyed by the resource fork's stream id). `None` for a directory, a symbolic link, an
/// empty file, or an inline-compressed file (whose content is in its attribute, already hashed as
/// part of the tree, not in a data stream). `read` returns a file's decoded content.
pub fn file_stream<E>(
    oid: u64,
    inode_val: &[u8],
    xattrs: &[(Vec<u8>, Vec<u8>)],
    read: impl FnOnce() -> Result<Vec<u8>, E>,
) -> Result<Option<(u64, Vec<u8>)>, E> {
    let Some(inode) = fs::Inode::parse(inode_val) else {
        return Ok(None);
    };
    if inode.mode & 0o170000 != 0o100000 {
        return Ok(None);
    }
    let compressed = inode.bsd_flags & crate::decmpfs::UF_COMPRESSED != 0;
    if !compressed {
        let content = read()?;
        return Ok((!content.is_empty()).then_some((oid, content)));
    }
    // A compressed file: its decmpfs header says whether its content is in the resource fork.
    let header = xattrs
        .iter()
        .find(|(k, _)| fs::xattr_name(k) == crate::decmpfs::XATTR)
        .and_then(|(_, v)| match fs::parse_xattr(v)?.1 {
            fs::XattrData::Embedded(d) => crate::decmpfs::Header::parse(&d),
            fs::XattrData::Stream(..) => None,
        });
    let Some(header) = header else {
        return Ok(None);
    };
    if !header.in_resource_fork() {
        return Ok(None);
    }
    let fork_id = xattrs
        .iter()
        .find(|(k, _)| fs::xattr_name(k) == crate::decmpfs::RESOURCE_FORK)
        .and_then(|(_, v)| match fs::parse_xattr(v)?.1 {
            fs::XattrData::Stream(id, _) => Some(id),
            fs::XattrData::Embedded(_) => None,
        });
    let Some(fork_id) = fork_id else {
        return Ok(None);
    };
    let content = read()?;
    Ok((!content.is_empty()).then_some((fork_id, content)))
}

/// The extents a file-extent record names, for building the physical file-extent tree the seal
/// does not cover: (stream id, logical offset, length, flags, physical block).
#[must_use]
pub fn file_extent(key: &[u8], val: &[u8]) -> (u64, u64, u64, u8, u64) {
    let stream = fs::key_header(key).0;
    let logical = u64_at(key, 8);
    let lf = u64_at(val, 0);
    (
        stream,
        logical,
        lf & 0x00ff_ffff_ffff_ffff,
        (lf >> 56) as u8,
        u64_at(val, 8),
    )
}

/// One data stream of a sealed volume: its identifier (an inode's private identifier, or an
/// attribute stream's), the inode that owns it, the blocks it is given (its allocated size, as
/// the seal records it), its size in bytes, and where its bytes are in the source volume.
#[derive(Clone, Debug)]
pub struct SealedStream {
    pub id: u64,
    pub owner: u64,
    pub blocks: u64,
    pub size: u64,
    /// The source's runs over the stream's `size` bytes, in order: `None` a hole.
    pub runs: Vec<crate::write::copy::SourceRun>,
}

/// An unsealed system volume made ready to seal: its canonical records in key order, the
/// `file_info` hashes among them, its data streams in the order of their first block in the
/// source, and what its superblock records.
pub struct SealedSet {
    pub records: Vec<Record>,
    pub streams: Vec<SealedStream>,
    pub counts: crate::write::volume::Counts,
    pub next_obj_id: u64,
    pub next_doc_id: u32,
    /// How many inodes share another's data stream (clones); the stream is given once.
    pub shared_streams: u64,
}

/// Read `vol`, an unsealed system volume, into the set its seal covers at mtree time `t`
/// (nanoseconds): every record canonicalised ([`canonicalize`]), each regular file's content
/// hashed into its `file_info` records ([`file_stream`], [`file_infos`]), and its data streams
/// gathered from the file extents the seal leaves out.
///
/// # Errors
///
/// The volume's records or a file's content fail their checks.
pub fn sealed_set(vol: &crate::Volume, t: u64) -> crate::Result<SealedSet> {
    use std::collections::HashMap;

    use crate::write::copy::SourceRun;

    let c = vol.container();
    let bs = u64::from(c.block_size());
    let source = vol.fs_tree_records()?;

    // Each stream's extents, each inode's value and attributes.
    let mut extents: HashMap<u64, Vec<fs::FileExtent>> = HashMap::new();
    let mut inodes: Vec<(u64, usize)> = Vec::new();
    let mut xattrs: HashMap<u64, Vec<usize>> = HashMap::new();
    for (i, (k, v)) in source.iter().enumerate() {
        let (oid, ty) = fs::key_header(k);
        match ty {
            rec::FILE_EXTENT => extents
                .entry(oid)
                .or_default()
                .push(fs::FileExtent::parse(k, v)),
            rec::INODE => inodes.push((oid, i)),
            rec::XATTR => xattrs.entry(oid).or_default().push(i),
            _ => {}
        }
    }
    for x in extents.values_mut() {
        x.sort_by_key(|e| e.logical);
    }
    let runs_of = |id: u64, size: u64| -> Vec<SourceRun> {
        let mut out = Vec::new();
        let mut at = 0u64;
        for e in extents.get(&id).map_or(&[][..], Vec::as_slice) {
            if e.logical >= size {
                break;
            }
            if e.logical > at {
                out.push(SourceRun {
                    len: e.logical - at,
                    phys: None,
                });
                at = e.logical;
            }
            let len = e.len.min(size - at);
            out.push(SourceRun {
                len,
                phys: (e.phys != 0).then_some(e.phys),
            });
            at += len;
        }
        if at < size {
            out.push(SourceRun {
                len: size - at,
                phys: None,
            });
        }
        out
    };
    let read_runs = |runs: &[SourceRun]| -> crate::Result<Vec<u8>> {
        let total: u64 = runs.iter().map(|r| r.len).sum();
        let mut out = vec![
            0u8;
            usize::try_from(total).map_err(|_| crate::Error::Unsupported(
                format!("a stream of {total} bytes")
            ))?
        ];
        let mut at = 0usize;
        for r in runs {
            let n = r.len as usize;
            if let Some(p) = r.phys {
                c.read_bytes(p, 0, &mut out[at..at + n])?;
            }
            at += n;
        }
        Ok(out)
    };

    // The streams: each inode's data stream and each attribute's, given once.
    let mut streams: HashMap<u64, SealedStream> = HashMap::new();
    let mut shared_streams = 0u64;
    for &(oid, i) in &inodes {
        let Some(inode) = fs::Inode::parse(&source[i].1) else {
            continue;
        };
        if let Some(d) = inode.dstream() {
            if streams.contains_key(&inode.private_id) {
                shared_streams += 1;
                continue;
            }
            streams.insert(
                inode.private_id,
                SealedStream {
                    id: inode.private_id,
                    owner: oid,
                    blocks: d.alloced_size.div_ceil(bs),
                    size: d.size,
                    runs: runs_of(inode.private_id, d.size),
                },
            );
        }
        for &x in xattrs.get(&oid).map_or(&[][..], Vec::as_slice) {
            if let Some((_, fs::XattrData::Stream(id, d))) = fs::parse_xattr(&source[x].1) {
                // The seal tightens an attribute stream's allocation to its size ([`canon_xattr`]).
                streams.entry(id).or_insert_with(|| SealedStream {
                    id,
                    owner: oid,
                    blocks: d.size.div_ceil(bs),
                    size: d.size,
                    runs: runs_of(id, d.size),
                });
            }
        }
    }
    let mut streams: Vec<SealedStream> = streams.into_values().collect();
    let first = |s: &SealedStream| s.runs.iter().find_map(|r| r.phys).unwrap_or(u64::MAX);
    streams.sort_by_key(|s| (first(s), s.id));

    // The file_info hashes, files read in the order of their first block in the source.
    let by_id: HashMap<u64, usize> = streams.iter().enumerate().map(|(n, s)| (s.id, n)).collect();
    let mut files: Vec<(u64, u64, usize)> = Vec::new();
    for &(oid, i) in &inodes {
        let Some(inode) = fs::Inode::parse(&source[i].1) else {
            continue;
        };
        if inode.mode & 0o170000 != 0o100000 {
            continue;
        }
        let at = by_id
            .get(&inode.private_id)
            .map_or(u64::MAX, |&n| first(&streams[n]));
        files.push((at, oid, i));
    }
    files.sort_unstable();
    let mut infos: Vec<Record> = Vec::new();
    let empty = Vec::new();
    for &(_, oid, i) in &files {
        let ival = &source[i].1;
        let xs: Vec<(Vec<u8>, Vec<u8>)> = xattrs
            .get(&oid)
            .unwrap_or(&empty)
            .iter()
            .map(|&x| source[x].clone())
            .collect();
        let inode = fs::Inode::parse(ival).expect("parsed above");
        let stream = file_stream(oid, ival, &xs, || -> crate::Result<Vec<u8>> {
            if inode.bsd_flags & crate::decmpfs::UF_COMPRESSED == 0 {
                let size = inode.dstream().map_or(0, |d| d.size);
                return read_runs(&runs_of(inode.private_id, size));
            }
            let attr = |name: &[u8]| {
                xs.iter()
                    .find(|(k, _)| fs::xattr_name(k) == name)
                    .and_then(|(_, v)| fs::parse_xattr(v))
                    .map(|x| x.1)
            };
            let Some(fs::XattrData::Embedded(header)) = attr(crate::decmpfs::XATTR) else {
                return Err(crate::corrupt(
                    0,
                    format!("compressed file {oid} without its header"),
                ));
            };
            let fork = match attr(crate::decmpfs::RESOURCE_FORK) {
                Some(fs::XattrData::Stream(id, d)) => Some(read_runs(&runs_of(id, d.size))?),
                Some(fs::XattrData::Embedded(b)) => Some(b),
                None => None,
            };
            crate::decmpfs::decode(&header, fork.as_deref())
        })?;
        if let Some((sid, content)) = stream {
            infos.extend(file_infos(sid, &content, bs as usize));
        }
    }

    // The canonical records, the hashes among them, in key order.
    let mut records: Vec<Record> = Vec::with_capacity(source.len() + infos.len());
    for (k, mut v) in source {
        let ty = fs::key_header(&k).1;
        match ty {
            rec::FILE_EXTENT => continue,
            rec::INODE => canon_inode(&mut v, t),
            rec::DIR_REC => canon_dir_rec(&mut v, t),
            rec::XATTR => canon_xattr(&mut v, bs),
            _ => {}
        }
        records.push(Record { key: k, val: v });
    }
    records.extend(infos);
    records.sort_by(|a, b| fs::compare_keys(&a.key, &b.key));

    let sb = &vol.sb;
    Ok(SealedSet {
        records,
        streams,
        counts: crate::write::volume::Counts {
            files: sb.num_files,
            directories: sb.num_directories,
            symlinks: sb.num_symlinks,
            other: sb.num_other_fsobjects,
        },
        next_obj_id: sb.next_obj_id,
        next_doc_id: sb.next_doc_id,
        shared_streams,
    })
}
