//! Writing the plan's three APFS containers into the disk, each at its partition: every volume
//! formed by reims-apfs's writer, and every file's bytes streamed from where the plan says they
//! come from — bytes the restore made, an IPSW member, an image's plaintext, or a boot object
//! stitched to its ticket (docs/design.md, "Writing the containers").

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use reims_apfs::write::{self as apfs_write, VolumeDef, copy, fsbuild, sealed, volume};
use uuid::Uuid;

use crate::aea::KeyService;
use crate::gpt::Partition;
use crate::image;
use crate::img4;
use crate::ipsw::Ipsw;
use crate::plan::{self, Content, Data, Node, Role};

/// Where the writer reads what it writes.
pub struct Sources<'a> {
    pub ipsw: &'a Ipsw,
    pub keys: &'a dyn KeyService,
    /// The system volume's root hash, once it is sealed: the boot directories' root hash
    /// objects are made from it.
    pub root_hash: Option<img4::RootHash>,
    /// When every volume records it was formatted and changed, and every entry made, in
    /// nanoseconds since 1970.
    pub t: u64,
}

/// One file's bytes, ready to stream into its blocks: held, or read front to back, `len` bytes.
enum Bytes {
    Held(Arc<[u8]>),
    Stream { len: u64, r: Box<dyn Read + Send> },
}

/// A copied stream's bytes, read run by run from the system image (a hole reads as zeros).
struct Runs {
    src: image::Source,
    block: u64,
    runs: Vec<copy::SourceRun>,
    at: usize,
    within: u64,
}

impl Read for Runs {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        while let Some(r) = self.runs.get(self.at) {
            if self.within == r.len {
                self.at += 1;
                self.within = 0;
                continue;
            }
            let n = (buf.len() as u64).min(r.len - self.within) as usize;
            match r.phys {
                Some(p) => self
                    .src
                    .read_exact_at(p * self.block + self.within, &mut buf[..n])?,
                None => buf[..n].fill(0),
            }
            self.within += n as u64;
            return Ok(n);
        }
        Ok(0)
    }
}

/// The Data volume's rule for the template's paths (docs/design.md, "The disk"): each pattern of
/// `Data.symlink-patterns` a glob from the template's root, `*` within one component and `**`
/// any depth; one ending in `/` matches directories, any other everything else, the first
/// match deciding. A match is made a symbolic link back into the template.
pub struct DataRule {
    rules: Vec<(bool, Vec<Tok>)>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Tok {
    Lit(u8),
    /// Any bytes within one component.
    Star,
    /// Any bytes at all.
    Any,
}

impl DataRule {
    /// The rule of the patterns, one per line.
    #[must_use]
    pub fn new(patterns: &str) -> Self {
        let rules = patterns
            .lines()
            .map(str::trim)
            .filter(|p| !p.is_empty())
            .map(|p| {
                let (dir, p) = p.strip_suffix('/').map_or((false, p), |d| (true, d));
                let mut toks = Vec::new();
                for (i, part) in p.split('/').enumerate() {
                    if i > 0 {
                        toks.push(Tok::Lit(b'/'));
                    }
                    if part == "**" {
                        toks.push(Tok::Any);
                        continue;
                    }
                    for b in part.bytes() {
                        toks.push(if b == b'*' { Tok::Star } else { Tok::Lit(b) });
                    }
                }
                (dir, toks)
            })
            .collect();
        Self { rules }
    }

    /// What becomes of the template's entry at `path` (from the template's root), a directory or
    /// not (a symbolic link is never asked): kept, or a link of `<../ per component of its
    /// parent>System/Library/Templates/Data/<path>`, a directory's ending in `/`.
    #[must_use]
    pub fn decide(&self, path: &str, is_dir: bool) -> copy::Replace {
        let hit = self
            .rules
            .iter()
            .any(|(dir, toks)| *dir == is_dir && matches(toks, path.as_bytes()));
        if !hit {
            return copy::Replace::Keep;
        }
        let mut target = "../".repeat(path.matches('/').count());
        target.push_str("System/Library/Templates/Data/");
        target.push_str(path);
        if is_dir {
            target.push('/');
        }
        copy::Replace::Link(target.into_bytes())
    }
}

/// Whether `toks` match all of `s`.
fn matches(toks: &[Tok], s: &[u8]) -> bool {
    match toks.split_first() {
        None => s.is_empty(),
        Some((Tok::Lit(c), rest)) => s.first() == Some(c) && matches(rest, &s[1..]),
        Some((Tok::Star, rest)) => {
            let span = s.iter().position(|&b| b == b'/').unwrap_or(s.len());
            (0..=span).any(|n| matches(rest, &s[n..]))
        }
        Some((Tok::Any, rest)) => (0..=s.len()).any(|n| matches(rest, &s[n..])),
    }
}

/// Where the data template is in the system image.
const TEMPLATE: &str = "/System/Library/Templates/Data";
/// Its patterns of what the Data volume links to instead of copying.
const PATTERNS: &str = "/System/Library/Templates/Data.symlink-patterns";
/// Where the Data volume keeps the template's receipt, and its flags (`SF_RESTRICTED`).
const RECEIPT: &str = "Library/Apple/System/Library/Receipts/com.apple.files.data-template.plist";
const SF_RESTRICTED: u32 = 0x8_0000;

/// The Data volume: the system image's data template, as [`DataRule`] makes it, with its
/// receipt.
fn data_volume<'a>(
    v: &'a plan::Volume,
    image_name: &str,
    receipt: &Arc<[u8]>,
    src: &Sources,
) -> Result<(VolumeDef<'a>, Vec<Bytes>)> {
    let (source, c) = image::open_container_source(src.ipsw, image_name, src.keys)?;
    let block = u64::from(c.block_size());
    let sys = c.system_volume()?;
    let root = sys
        .lookup(TEMPLATE)
        .with_context(|| format!("find {TEMPLATE} in {image_name}"))?;
    let rule = DataRule::new(&String::from_utf8_lossy(&sys.read_file(PATTERNS)?));
    let records = sys
        .fs_tree_records()
        .context("read the system image's records")?;
    let added = [fsbuild::Item {
        path: RECEIPT.into(),
        kind: fsbuild::Kind::File {
            len: receipt.len() as u64,
        },
        mode: 0o666,
        uid: 0,
        gid: 0,
        bsd_flags: SF_RESTRICTED,
        xattrs: Vec::new(),
    }];
    let copied = copy::copy_tree(
        &records,
        root.id,
        true,
        src.t,
        &mut |p, d| rule.decide(p, d),
        &added,
    )
    .context("copy the data template")?;
    drop(records);
    let bytes = copied
        .sources
        .iter()
        .zip(&copied.streams)
        .map(|(s, st)| match s {
            copy::StreamSource::Runs(runs) => Bytes::Stream {
                len: st.len,
                r: Box::new(Runs {
                    src: source.clone(),
                    block,
                    runs: runs.clone(),
                    at: 0,
                    within: 0,
                }),
            },
            copy::StreamSource::Added(_) => Bytes::Held(receipt.clone()),
        })
        .collect();
    let def = VolumeDef {
        name: v.name,
        uuid: *v.uuid.as_bytes(),
        role: v.role.apfs(),
        system: false,
        group: v.group.map_or([0; 16], |g| *g.as_bytes()),
        records: copied.records,
        streams: copied.streams,
        counts: copied.counts,
        next_obj_id: copied.next_obj_id,
        seal: None,
    };
    Ok((def, bytes))
}

/// The system volume: the OS image's volume sealed with reims-apfs's own tree at the time its
/// canonical metadata `mtree` records (docs/design.md, "The system volume's seal"). Sets
/// `src.root_hash`, whose first hash is the seal's and the other two the IPSW's `root_hash`'s.
fn system_volume<'a>(
    v: &'a plan::Volume,
    image_name: &str,
    mtree: &str,
    root_hash: &str,
    src: &mut Sources,
) -> Result<(VolumeDef<'a>, Vec<Bytes>)> {
    let t = crate::mtree::root_time(&src.ipsw.read(mtree)?)
        .with_context(|| format!("read the seal's time from {mtree}"))?;
    let apple = img4::parse_im4p(&src.ipsw.read(root_hash)?)
        .and_then(|p| img4::RootHash::parse(&p.payload))
        .with_context(|| format!("read {root_hash}"))?;
    let (source, c) = image::open_container_source(src.ipsw, image_name, src.keys)?;
    let block = u64::from(c.block_size());
    ensure!(
        block == apfs_write::geometry::BLOCK_SIZE,
        "{image_name}'s blocks are {block} bytes"
    );
    let sys = c.system_volume()?;
    let set = reims_apfs::canon::sealed_set(&sys, t)
        .with_context(|| format!("read {image_name}'s system volume"))?;
    let tree = reims_apfs::seal::build(&set.records, 0);
    let reims_apfs::canon::SealedSet {
        records,
        streams,
        counts,
        next_obj_id,
        next_doc_id,
        ..
    } = set;
    drop(records);
    let hash = img4::RootHash {
        hashes: [tree.root_hash, apple.hashes[1], apple.hashes[2]],
    };
    let seal = sealed::Seal {
        tree,
        streams: streams
            .iter()
            .map(|s| volume::Stream {
                id: s.id,
                owner: s.owner,
                len: s.blocks * block,
            })
            .collect(),
        snapshot: hash.snapshot_name(),
        snapshot_uuid: *Uuid::new_v4().as_bytes(),
        inum: next_obj_id,
        next_doc_id,
    };
    src.root_hash = Some(hash);
    // Each stream's bytes, and zeros to the end of its blocks: a sealed file's last segment is
    // hashed zero-padded to whole blocks.
    let bytes = streams
        .into_iter()
        .map(|s| {
            let len = s.blocks * block;
            ensure!(
                s.size <= len,
                "stream {} holds {} bytes in {len}",
                s.id,
                s.size
            );
            let mut runs = s.runs;
            if len > s.size {
                runs.push(copy::SourceRun {
                    len: len - s.size,
                    phys: None,
                });
            }
            Ok(Bytes::Stream {
                len,
                r: Box::new(Runs {
                    src: source.clone(),
                    block,
                    runs,
                    at: 0,
                    within: 0,
                }),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let group = v.group.map_or([0; 16], |g| *g.as_bytes());
    let def = VolumeDef::sealed(v.name, *v.uuid.as_bytes(), group, counts, seal);
    Ok((def, bytes))
}

/// A file's length and where its bytes come from.
fn open_data(data: &Data, src: &Sources) -> Result<(u64, Bytes)> {
    Ok(match data {
        Data::Bytes(b) => (b.len() as u64, Bytes::Held(b.clone())),
        Data::Member(m) => {
            let len = src.ipsw.member(m)?.size;
            (
                len,
                Bytes::Stream {
                    len,
                    r: src.ipsw.reader(m)?,
                },
            )
        }
        Data::Plain(m) => {
            let (len, r) = image::plain_stream(src.ipsw, m, src.keys)?;
            (len, Bytes::Stream { len, r })
        }
        Data::Stitched {
            component,
            member,
            manifest,
        } => {
            let out = img4::personalize(component, &src.ipsw.read(member)?, manifest)
                .with_context(|| format!("stitch {component} from {member}"))?;
            (out.len() as u64, Bytes::Held(out.into()))
        }
        Data::SealedRootHash { manifest } => {
            let seal = src
                .root_hash
                .as_ref()
                .context("a boot directory's root hash before the system volume is sealed")?;
            let out = img4::stitch_img4(&seal.system_im4p(), manifest);
            (out.len() as u64, Bytes::Held(out.into()))
        }
    })
}

/// A volume of the plan's entries: the writer's description of it, and the bytes of each of its
/// data streams, in the order of the streams.
fn entries_volume<'a>(
    v: &'a plan::Volume,
    entries: &[plan::Entry],
    src: &Sources,
) -> Result<(VolumeDef<'a>, Vec<Bytes>)> {
    let mut items = Vec::with_capacity(entries.len());
    let mut bytes: Vec<Option<Bytes>> = Vec::with_capacity(entries.len());
    for e in entries {
        let (kind, b) = match &e.node {
            Node::Dir => (fsbuild::Kind::Dir, None),
            Node::File(d) => {
                let (len, b) =
                    open_data(d, src).with_context(|| format!("{}/{}", v.name, e.path))?;
                (fsbuild::Kind::File { len }, Some(b))
            }
            Node::Symlink(t) => (fsbuild::Kind::Symlink(t.as_bytes().to_vec()), None),
            Node::Clone(of) => (fsbuild::Kind::Clone(of.clone()), None),
        };
        items.push(fsbuild::Item {
            path: e.path.clone(),
            kind,
            mode: e.mode,
            uid: e.uid,
            gid: e.gid,
            bsd_flags: 0,
            xattrs: Vec::new(),
        });
        bytes.push(b);
    }
    let built =
        fsbuild::build(&items, true, src.t).with_context(|| format!("lay out {}", v.name))?;
    let streams = built
        .stream_items
        .iter()
        .map(|&i| {
            bytes[i]
                .take()
                .context("a stream for an entry without bytes")
        })
        .collect::<Result<Vec<_>>>()?;
    let mut def = VolumeDef::of(v.name, *v.uuid.as_bytes(), v.role.apfs(), built);
    def.system = v.role == Role::System;
    def.group = v.group.map_or([0; 16], |g| *g.as_bytes());
    Ok((def, streams))
}

/// Copy exactly `len` bytes from `r` to `w`.
fn copy_exact(r: &mut dyn Read, w: &mut File, len: u64) -> Result<()> {
    let mut buf = vec![0u8; 1 << 20];
    let mut left = len;
    while left > 0 {
        let n = (buf.len() as u64).min(left) as usize;
        r.read_exact(&mut buf[..n])?;
        w.write_all(&buf[..n])?;
        left -= n as u64;
    }
    // Nothing past the length the stream gave.
    let mut probe = [0u8; 1];
    ensure!(
        matches!(r.read(&mut probe), Ok(0)),
        "a stream runs past the {len} bytes it gave"
    );
    Ok(())
}

/// Write the container of `volumes` at `part` of `disk`. Sealing a system volume sets
/// `src.root_hash`, which the volumes after it that hold boot directories need.
///
/// # Errors
///
/// An entry the writer cannot lay out, a source that cannot be read, volumes that do not fit
/// the partition, or a write that fails.
pub fn write_container(
    disk: &mut File,
    part: &Partition,
    volumes: &[plan::Volume],
    src: &mut Sources,
) -> Result<()> {
    let block = apfs_write::geometry::BLOCK_SIZE;
    let block_count = part.len() / block;
    let mut defs = Vec::with_capacity(volumes.len());
    let mut all_bytes = Vec::with_capacity(volumes.len());
    for v in volumes {
        match &v.content {
            Content::Entries(entries) => {
                let (def, bytes) = entries_volume(v, entries, src)?;
                defs.push(def);
                all_bytes.push(bytes);
            }
            Content::DataTemplate { image, receipt } => {
                let (def, bytes) = data_volume(v, image, receipt, src)
                    .with_context(|| format!("lay out {}", v.name))?;
                defs.push(def);
                all_bytes.push(bytes);
            }
            Content::SealedSystem {
                image,
                mtree,
                root_hash,
            } => {
                let (def, bytes) = system_volume(v, image, mtree, root_hash, src)
                    .with_context(|| format!("seal {}", v.name))?;
                defs.push(def);
                all_bytes.push(bytes);
            }
        }
    }
    let formed = apfs_write::container(block_count, *Uuid::new_v4().as_bytes(), src.t, defs)
        .context("lay out the container")?;
    let base = part.offset();
    for (addr, b) in &formed.blocks {
        disk.seek(SeekFrom::Start(base + addr * block))?;
        disk.write_all(b)?;
    }
    for ((v, firsts), bytes) in volumes.iter().zip(&formed.stream_blocks).zip(all_bytes) {
        for (first, b) in firsts.iter().zip(bytes) {
            disk.seek(SeekFrom::Start(base + first * block))?;
            match b {
                Bytes::Held(h) => disk.write_all(&h)?,
                // A stream that gives other than the length the writer laid out fails here.
                Bytes::Stream { len, mut r } => copy_exact(&mut r, disk, len)
                    .with_context(|| format!("write {}'s files", v.name))?,
            }
        }
    }
    disk.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn link(r: copy::Replace) -> Option<String> {
        match r {
            copy::Replace::Keep => None,
            copy::Replace::Link(t) => Some(String::from_utf8(t).unwrap()),
        }
    }

    // Patterns of the forms the data templates hold, written here: a directory pattern with `*`
    // in a component, a file pattern with `*` inside a component, and `**` at the end.
    const PATTERNS: &str = "\
Library/Things/*.bundle/Contents/
System/Assets/*/kind_A/*.asset/Data/
System/Core.bundle/Contents/Icons*.bundle/Res/*.icns
Library/Pictures/**

";

    #[test]
    fn a_directory_pattern_links_directories_only() {
        let r = DataRule::new(PATTERNS);
        assert_eq!(
            link(r.decide("Library/Things/x.bundle/Contents", true)).as_deref(),
            Some("../../../System/Library/Templates/Data/Library/Things/x.bundle/Contents/")
        );
        assert_eq!(
            link(r.decide("Library/Things/x.bundle/Contents", false)),
            None
        );
        assert_eq!(link(r.decide("Library/Things/x.bundle/Other", true)), None);
        // `*` stays within one component.
        assert_eq!(
            link(r.decide("Library/Things/a/b.bundle/Contents", true)),
            None
        );
        assert_eq!(
            link(r.decide("System/Assets/v1/kind_A/q.asset/Data", true)).as_deref(),
            Some(
                "../../../../../System/Library/Templates/Data/System/Assets/v1/kind_A/q.asset/Data/"
            )
        );
    }

    #[test]
    fn a_file_pattern_links_files_and_keeps_directories() {
        let r = DataRule::new(PATTERNS);
        assert_eq!(
            link(r.decide(
                "System/Core.bundle/Contents/IconsX.bundle/Res/a.icns",
                false
            ))
            .as_deref(),
            Some(
                "../../../../../System/Library/Templates/Data/System/Core.bundle/Contents/IconsX.bundle/Res/a.icns"
            )
        );
        assert!(
            link(r.decide("System/Core.bundle/Contents/Icons.bundle/Res/a.icns", false)).is_some()
        );
        assert_eq!(
            link(r.decide("System/Core.bundle/Contents/Icons.bundle/Res/a.png", false)),
            None
        );
        // `**` links every file below, at any depth, and keeps the directories.
        assert_eq!(
            link(r.decide("Library/Pictures/Animals/Cat.heic", false)).as_deref(),
            Some("../../../System/Library/Templates/Data/Library/Pictures/Animals/Cat.heic")
        );
        assert_eq!(link(r.decide("Library/Pictures/Animals", true)), None);
        assert_eq!(link(r.decide("Library/Pictures", true)), None);
        assert_eq!(link(r.decide("Library/Other/Cat.heic", false)), None);
    }
}
