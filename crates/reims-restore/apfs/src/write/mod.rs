//! Writing APFS: building a container's structures block by block, from the file-system records
//! up, as a restore writes a guest's disk without any of Apple's tools.
//!
//! The pieces are built bottom-up. [`node`] serialises one B-tree node; [`records`] builds the
//! file-system records a volume holds. Higher layers — the volume's trees, the container's
//! space manager and object map, and the whole disk — are added as the writer grows.

pub mod copy;
pub mod fsbuild;
pub mod geometry;
pub mod node;
pub mod records;
pub mod sealed;
pub mod space;
pub mod tree;
pub mod volume;

use crate::ondisk::apfs;

/// A bare container of `block_count` 4 KiB blocks, identified by `uuid`, holding no volume — the
/// state a freshly formatted container is in before a volume is added. Returns the blocks it
/// writes, as (block address, bytes); every other block of the container is zero.
#[must_use]
pub fn bare_container(block_count: u64, uuid: [u8; 16]) -> Vec<(u64, Vec<u8>)> {
    // Past the internal pool the container object map takes two blocks, its object and its tree.
    let g = geometry::Geometry::of(block_count);
    space::scaffold(&g, uuid, &[], &[], space::FIRST_VOLUME_OID, 2)
}

/// A volume to place in a container: its name, identifier, role and group, its features, and
/// its file-system records (but file extents) and data streams, as [`fsbuild::build`] makes them
/// from a tree of files.
pub struct VolumeDef<'a> {
    pub name: &'a str,
    pub uuid: [u8; 16],
    pub role: u16,
    /// Whether it is a volume group's System volume, which carries the group feature.
    pub system: bool,
    /// The volume group's identifier, or zero.
    pub group: [u8; 16],
    pub records: Vec<(Vec<u8>, Vec<u8>)>,
    pub streams: Vec<volume::Stream>,
    pub counts: volume::Counts,
    pub next_obj_id: u64,
    /// A sealed volume's seal, in place of `records` and `streams`.
    pub seal: Option<sealed::Seal>,
}

impl<'a> VolumeDef<'a> {
    /// A volume holding the tree `built`.
    #[must_use]
    pub fn of(name: &'a str, uuid: [u8; 16], role: u16, built: fsbuild::Built) -> Self {
        Self {
            name,
            uuid,
            role,
            system: false,
            group: [0; 16],
            records: built.records,
            streams: built.streams,
            counts: built.counts,
            next_obj_id: built.next_obj_id,
            seal: None,
        }
    }

    /// A sealed system volume of `seal`, in volume group `group`.
    #[must_use]
    pub fn sealed(
        name: &'a str,
        uuid: [u8; 16],
        group: [u8; 16],
        counts: volume::Counts,
        seal: sealed::Seal,
    ) -> Self {
        Self {
            name,
            uuid,
            role: apfs::ROLE_SYSTEM,
            system: true,
            group,
            records: Vec::new(),
            streams: Vec::new(),
            counts,
            next_obj_id: seal.inum + 1,
            seal: Some(seal),
        }
    }
}

/// A formed container: the blocks of its structures (block, bytes), every other block zero, and
/// for each volume the first block of each of its streams, where the caller writes the stream's
/// bytes (0 for an empty one).
pub struct Formed {
    pub blocks: Vec<(u64, Vec<u8>)>,
    pub stream_blocks: Vec<Vec<u64>>,
}

/// A container of `block_count` 4 KiB blocks holding `volumes`, each placed past the container's
/// object map in the order given, and recording `t` (nanoseconds since 1970) as when each volume
/// was formatted and last changed.
///
/// # Errors
///
/// [`tree::TreeError`] when the volumes do not fit the container.
pub fn container(
    block_count: u64,
    uuid: [u8; 16],
    t: u64,
    volumes: Vec<VolumeDef>,
) -> Result<Formed, tree::TreeError> {
    // A container is given at least as many volume slots as it holds, which sizes its checkpoint
    // areas; one geometry is shared by the scaffold and the volume placement.
    let max_volumes = geometry::volume_capacity(block_count).max(volumes.len() as u64);
    let g = geometry::Geometry::with_max_volumes(block_count, max_volumes);
    // The container object map takes the first two blocks past the pool; the volumes follow.
    let mut alloc = tree::Alloc::new(g.first_free + 2, block_count);
    let mut next_oid = space::FIRST_VOLUME_OID;
    let mut blocks: Vec<(u64, Vec<u8>)> = Vec::new();
    let mut omap_entries: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
    let mut fs_oids: Vec<u64> = Vec::new();
    let mut stream_blocks = Vec::new();

    for (i, def) in volumes.into_iter().enumerate() {
        let features = if def.system { 0x12 } else { 0x2 };
        if let Some(seal) = def.seal {
            let built = sealed::sealed_volume(
                sealed::SealedSpec {
                    index: i as u32,
                    name: def.name,
                    uuid: def.uuid,
                    role: def.role,
                    features,
                    group: def.group,
                    t,
                    counts: def.counts,
                    seal,
                },
                &mut alloc,
                &mut next_oid,
            )?;
            blocks.extend(built.blocks);
            omap_entries.push(built.container_omap_record);
            fs_oids.push(built.sb_oid);
            stream_blocks.push(built.stream_blocks);
            continue;
        }
        let built = volume::volume(
            volume::VolumeSpec {
                index: i as u32,
                name: def.name,
                uuid: def.uuid,
                role: def.role,
                features,
                incompat: apfs::INCOMPAT_CASE_INSENSITIVE,
                next_obj_id: def.next_obj_id,
                group: def.group,
                t,
                records: def.records,
                streams: &def.streams,
                counts: def.counts,
            },
            &mut alloc,
            &mut next_oid,
        )?;
        blocks.extend(built.blocks);
        omap_entries.push(built.container_omap_record);
        fs_oids.push(built.sb_oid);
        stream_blocks.push(built.stream_blocks);
    }

    // The container object map's records order by their two-word key (identifier, transaction).
    omap_entries.sort_by(|a, b| crate::fs::compare_u64_pairs(&a.0, &b.0));
    let used_after_pool = alloc.next() - g.first_free;
    let mut out = space::scaffold(&g, uuid, &omap_entries, &fs_oids, next_oid, used_after_pool);
    out.extend(blocks);
    Ok(Formed {
        blocks: out,
        stream_blocks,
    })
}

/// A fixed instant in 2026 (nanoseconds since 1970) for the writer's tests and tools.
pub const FORMAT_TIME: u64 = 1_793_000_000_000_000_000;

/// Write `blocks` (block address, bytes) into `sink`, a destination already the container's full
/// size and reading back as zero where nothing is written, so it stays sparse.
///
/// # Errors
///
/// Any write or seek error from `sink`.
pub fn write_blocks<W: std::io::Write + std::io::Seek>(
    sink: &mut W,
    blocks: &[(u64, Vec<u8>)],
) -> std::io::Result<()> {
    for (addr, bytes) in blocks {
        sink.seek(std::io::SeekFrom::Start(addr * geometry::BLOCK_SIZE))?;
        sink.write_all(bytes)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::node::{Layout, NodeSpec};
    use super::records::{self, DT_DIR, InodeBuilder, name_xfield};
    use std::collections::HashMap;
    use std::sync::Arc;

    /// A container held as only the blocks written to it, the rest zero — so a reader touches a
    /// handful of blocks without the whole container existing in memory.
    struct SparseBlocks {
        blocks: HashMap<u64, Vec<u8>>,
        len: u64,
    }

    impl SparseBlocks {
        fn of(block_count: u64, written: Vec<(u64, Vec<u8>)>) -> Self {
            Self {
                blocks: written.into_iter().collect(),
                len: block_count * super::geometry::BLOCK_SIZE,
            }
        }
    }

    impl crate::dev::Source for SparseBlocks {
        fn read_at(&self, off: u64, buf: &mut [u8]) -> std::io::Result<()> {
            let bs = super::geometry::BLOCK_SIZE as usize;
            let mut done = 0usize;
            while done < buf.len() {
                let pos = off as usize + done;
                let (blk, within) = ((pos / bs) as u64, pos % bs);
                let n = (bs - within).min(buf.len() - done);
                match self.blocks.get(&blk) {
                    Some(data) => buf[done..done + n].copy_from_slice(&data[within..within + n]),
                    None => buf[done..done + n].fill(0),
                }
                done += n;
            }
            Ok(())
        }
        fn len(&self) -> u64 {
            self.len
        }
    }

    // A bare container the writer builds opens as an APFS container with no volumes, its object
    // map and space manager read and checked by the reader (checksums, identifiers, transactions).
    #[test]
    fn a_bare_container_opens_and_reads() {
        let block_count = 128_000; // the iBoot system container's size
        let blocks = super::bare_container(block_count, [0x33; 16]);
        let src = Arc::new(SparseBlocks::of(block_count, blocks));
        let c = crate::open(src).expect("open the bare container");
        assert_eq!(c.volumes().expect("list volumes").len(), 0);
    }

    // A container with two empty volumes opens, lists them by name, and reads each one's
    // file-system records (its root and private directories) back through the reader.
    #[test]
    fn a_container_with_volumes_opens_and_reads() {
        let block_count = 128_000;
        let empty = || super::fsbuild::build(&[], true, super::FORMAT_TIME).unwrap();
        let mut system = super::VolumeDef::of(
            "Macintosh HD",
            [0x11; 16],
            crate::ondisk::apfs::ROLE_SYSTEM,
            empty(),
        );
        system.system = true;
        let vols = vec![
            system,
            super::VolumeDef::of(
                "Preboot",
                [0x22; 16],
                crate::ondisk::apfs::ROLE_PREBOOT,
                empty(),
            ),
        ];
        let formed = super::container(block_count, [0x33; 16], super::FORMAT_TIME, vols).unwrap();
        let src = Arc::new(SparseBlocks::of(block_count, formed.blocks));
        let c = crate::open(src).expect("open the container");
        let names: Vec<String> = c
            .volumes()
            .expect("list volumes")
            .iter()
            .map(|v| v.name().to_owned())
            .collect();
        assert_eq!(names, ["Macintosh HD", "Preboot"]);
        let recs = c.volumes().unwrap()[1]
            .fs_tree_records()
            .expect("read Preboot's records");
        assert_eq!(recs.len(), 4); // two directory entries and two inodes
    }

    /// A tree of directories, files of many sizes, empty files, symbolic links, attributes and a
    /// clone, big enough that every tree of the volume has index nodes, and the bytes of each
    /// file.
    pub(super) fn many_files(n: usize) -> (Vec<super::fsbuild::Item>, Vec<Vec<u8>>) {
        use super::fsbuild::{Item, Kind};
        let item = |path: String, kind: Kind, mode: u16| Item {
            path,
            kind,
            mode,
            uid: 0,
            gid: 0,
            bsd_flags: 0,
            xattrs: Vec::new(),
        };
        let mut items = Vec::new();
        let mut bytes = Vec::new();
        let dir = |items: &mut Vec<Item>, bytes: &mut Vec<Vec<u8>>, p: &str| {
            items.push(item(p.to_owned(), Kind::Dir, 0o755));
            bytes.push(Vec::new());
        };
        dir(&mut items, &mut bytes, "top");
        for d in 0..8 {
            dir(&mut items, &mut bytes, &format!("top/sub{d}"));
        }
        for i in 0..n {
            let len = (i * 997) % 20_000;
            let body: Vec<u8> = (0..len).map(|j| (i + j * 31) as u8).collect();
            let mut it = item(
                format!("top/sub{}/file-{i:05}", i % 8),
                Kind::File { len: len as u64 },
                0o644,
            );
            if i % 50 == 0 {
                it.xattrs
                    .push((b"com.example.tag".to_vec(), format!("tag {i}").into_bytes()));
            }
            items.push(it);
            bytes.push(body);
            if i % 100 == 0 {
                items.push(item(
                    format!("top/link-{i:05}"),
                    Kind::Symlink(format!("sub{}/file-{i:05}", i % 8).into_bytes()),
                    0o755,
                ));
                bytes.push(Vec::new());
            }
        }
        // A clone of a file with bytes.
        let orig = items
            .iter()
            .position(|it| matches!(it.kind, Kind::File { len } if len > 0))
            .unwrap();
        let orig_path = items[orig].path.clone();
        let orig_bytes = bytes[orig].clone();
        items.push(item("top/the-clone".into(), Kind::Clone(orig_path), 0o644));
        bytes.push(orig_bytes);
        (items, bytes)
    }

    /// Form a container of one volume holding `items`, writing each stream's bytes, as a sparse
    /// set of blocks.
    pub(super) fn form_with_files(
        block_count: u64,
        items: &[super::fsbuild::Item],
        bytes: &[Vec<u8>],
    ) -> HashMap<u64, Vec<u8>> {
        let built = super::fsbuild::build(items, true, super::FORMAT_TIME).unwrap();
        let stream_items = built.stream_items.clone();
        let def = super::VolumeDef::of("Files", [0x44; 16], crate::ondisk::apfs::ROLE_NONE, built);
        let formed =
            super::container(block_count, [0x55; 16], super::FORMAT_TIME, vec![def]).unwrap();
        let mut blocks: HashMap<u64, Vec<u8>> = formed.blocks.into_iter().collect();
        for (s, &first) in formed.stream_blocks[0].iter().enumerate() {
            let data = &bytes[stream_items[s]];
            for (k, chunk) in data.chunks(4096).enumerate() {
                let mut b = chunk.to_vec();
                b.resize(4096, 0);
                blocks.insert(first + k as u64, b);
            }
        }
        blocks
    }

    // A volume of thousands of files reads back through the reader: every file's bytes, every
    // link's target, the clone's bytes, and the attributes.
    #[test]
    fn a_volume_of_many_files_reads_back() {
        use super::fsbuild::Kind;
        let block_count = 128_000;
        let (items, bytes) = many_files(3000);
        let blocks = form_with_files(block_count, &items, &bytes);
        let src = Arc::new(SparseBlocks {
            blocks,
            len: block_count * super::geometry::BLOCK_SIZE,
        });
        let c = crate::open(src).expect("open the container");
        let vols = c.volumes().unwrap();
        let v = &vols[0];
        let shape = v.fs_tree_shape().unwrap();
        assert!(
            shape.iter().any(|n| n.level > 0),
            "the file-system tree has index nodes"
        );
        for (it, want) in items.iter().zip(&bytes) {
            let path = format!("/{}", it.path);
            match &it.kind {
                Kind::File { .. } | Kind::Clone(_) => {
                    let got = v.read_file(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
                    assert!(
                        got == *want,
                        "{path}: {} bytes read, {} written",
                        got.len(),
                        want.len()
                    );
                }
                _ => {}
            }
        }
    }

    // A sealed system volume of thousands of files reads back through its seal: the reader
    // checks every node of the hashed tree against its parent's entry and the root against the
    // integrity metadata, and reads every file's bytes through the file-extent tree.
    #[test]
    fn a_sealed_volume_reads_back_through_its_seal() {
        use super::fsbuild::Kind;
        use super::sealed::Seal;
        use super::volume::Stream;
        use crate::seal::{self, Record};
        let block_count = 128_000;
        let (all, all_bytes) = many_files(3000);
        // A sealed set gives a cloned stream once; this volume holds none.
        let keep: Vec<usize> = (0..all.len())
            .filter(|&i| !matches!(all[i].kind, Kind::Clone(_)))
            .collect();
        let items: Vec<_> = keep.iter().map(|&i| all[i].clone()).collect();
        let bytes: Vec<_> = keep.iter().map(|&i| all_bytes[i].clone()).collect();
        let built = super::fsbuild::build(&items, true, super::FORMAT_TIME).unwrap();
        let mut recs: Vec<Record> = built
            .records
            .iter()
            .map(|(k, v)| Record {
                key: k.clone(),
                val: v.clone(),
            })
            .collect();
        for (st, &it) in built.streams.iter().zip(&built.stream_items) {
            recs.extend(crate::canon::file_infos(st.id, &bytes[it], 4096));
        }
        recs.sort_by(|a, b| crate::fs::compare_keys(&a.key, &b.key));
        let tree = seal::build(&recs, 0);
        let hash = tree.root_hash;
        assert!(
            tree.nodes.iter().any(|n| n.level > 1),
            "the hashed tree has two index levels"
        );
        let streams = built
            .streams
            .iter()
            .map(|s| Stream {
                id: s.id,
                owner: s.owner,
                len: s.len.div_ceil(4096) * 4096,
            })
            .collect();
        let seal = Seal {
            tree,
            streams,
            snapshot: "com.apple.os.update-test".into(),
            snapshot_uuid: [0x66; 16],
            inum: built.next_obj_id,
            next_doc_id: 3,
        };
        let def =
            super::VolumeDef::sealed("Macintosh HD", [0x44; 16], [0x77; 16], built.counts, seal);
        let formed =
            super::container(block_count, [0x55; 16], super::FORMAT_TIME, vec![def]).unwrap();
        let mut blocks: HashMap<u64, Vec<u8>> = formed.blocks.into_iter().collect();
        for (s, &first) in formed.stream_blocks[0].iter().enumerate() {
            for (k, chunk) in bytes[built.stream_items[s]].chunks(4096).enumerate() {
                let mut b = chunk.to_vec();
                b.resize(4096, 0);
                blocks.insert(first + k as u64, b);
            }
        }
        let c = crate::open(Arc::new(SparseBlocks {
            blocks,
            len: block_count * super::geometry::BLOCK_SIZE,
        }))
        .expect("open the container");
        let v = c.system_volume().unwrap();
        assert_eq!(
            v.root_hash(),
            Some(&hash[..]),
            "the integrity metadata's root hash"
        );
        let got = v
            .fs_tree_records()
            .expect("read the hashed tree, every node checked");
        assert_eq!(got.len(), recs.len());
        assert!(
            got.iter()
                .zip(&recs)
                .all(|((k, v), r)| *k == r.key && *v == r.val)
        );
        for (it, want) in items.iter().zip(&bytes) {
            if let Kind::File { .. } = it.kind {
                let path = format!("/{}", it.path);
                let got = v.read_file(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
                assert!(
                    got == *want,
                    "{path}: {} bytes read, {} written",
                    got.len(),
                    want.len()
                );
            }
        }
        let facts = v.seal_facts().unwrap();
        assert!(
            facts.fext_root_has_object_header,
            "the file-extent tree is outside the seal"
        );
        assert!(
            !facts
                .fs_record_types
                .contains_key(&crate::ondisk::rec::FILE_EXTENT)
        );
    }

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    // The four records of a freshly made volume, read off a container `diskutil` formatted
    // (`/tmp/rvg-r1/ref-fresh/vols.raw`, volume 0's file-system tree). Times differ between the
    // root and the private directory, so each record carries its own.
    const ROOT_T: u64 = 0x18e7_3f55_80a3_e8bb;
    const PRIV_T: u64 = 0x18e7_3f55_80a3_fd13;

    #[test]
    fn a_directory_entry_matches_a_formatted_volumes() {
        let (k, v) = records::dir_rec(1, b"root", true, 2, ROOT_T, DT_DIR);
        assert_eq!(k, unhex("010000000000009005e471b6726f6f7400"));
        assert_eq!(v, unhex("0200000000000000bbe8a380553fe7180400"));
        let (k, v) = records::dir_rec(1, b"private-dir", true, 3, PRIV_T, DT_DIR);
        assert_eq!(k, unhex("01000000000000900c8ca6ac707269766174652d64697200"));
        assert_eq!(v, unhex("030000000000000013fda380553fe7180400"));
    }

    fn fresh_dir(oid: u64, name: &[u8], t: u64, mode: u16) -> InodeBuilder {
        InodeBuilder {
            oid,
            parent: 1,
            private_id: oid,
            create_time: t,
            mod_time: t,
            change_time: t,
            access_time: t,
            internal_flags: records::INODE_FLAGS_BASE,
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
    }

    #[test]
    fn an_inode_matches_a_formatted_volumes() {
        let (k, v) = fresh_dir(2, b"root", ROOT_T, 0o040755).build();
        assert_eq!(k, unhex("0200000000000030"));
        assert_eq!(
            v,
            unhex(
                "01000000000000000200000000000000bbe8a380553fe718bbe8a380553fe718bbe8a380553fe718\
                 bbe8a380553fe7180080000000000000000000000000000001000000000000000000000000000000\
                 ed41000000000000000000000100080004020500726f6f7400000000"
            )
        );
        let (k, v) = fresh_dir(3, b"private-dir", PRIV_T, 0o040644).build();
        assert_eq!(k, unhex("0300000000000030"));
        assert_eq!(
            v,
            unhex(
                "0100000000000000030000000000000013fda380553fe71813fda380553fe71813fda380553fe718\
                 13fda380553fe7180080000000000000000000000000000001000000000000000000000000000000\
                 a441000000000000000000000100100004020c00707269766174652d6469720000000000"
            )
        );
    }

    // A root-and-leaf node of the four records lays out exactly as the formatted volume's does:
    // table of contents 64 bytes, keys 57, the free gap, then the values, and a valid checksum.
    #[test]
    fn an_empty_volume_node_lays_out_as_a_formatted_ones() {
        let recs = records::empty_volume_records(true, ROOT_T);
        let longest_key = recs.iter().map(|(k, _)| k.len()).max().unwrap() as u32;
        let longest_val = recs.iter().map(|(_, v)| v.len()).max().unwrap() as u32;
        let node = NodeSpec {
            obj_type: 0x0000_0002, // virtual B-tree
            subtype: 0x0e,         // file-system tree
            oid: 0x404,
            xid: 2,
            level: 0,
            is_root: true,
            headerless: false,
            hashed: false,
            layout: Layout::Variable,
            tree_flags: 0x42,
            longest_key,
            longest_val,
            key_count: recs.len() as u64,
            node_count: 1,
            entries: &recs,
        }
        .build();

        assert_eq!(node.len(), 4096);
        assert!(crate::fletcher::verify(&node));
        // Header, flags, level and key count.
        assert_eq!(u32::from_le_bytes(node[24..28].try_into().unwrap()), 0x02);
        assert_eq!(u32::from_le_bytes(node[28..32].try_into().unwrap()), 0x0e);
        assert_eq!(u16::from_le_bytes(node[32..34].try_into().unwrap()), 0x3); // root|leaf
        assert_eq!(u16::from_le_bytes(node[34..36].try_into().unwrap()), 0); // level
        assert_eq!(u32::from_le_bytes(node[36..40].try_into().unwrap()), 4); // nkeys
        // Table of contents: offset 0, length 64. Free gap starts at 57 (the key bytes).
        assert_eq!(u16::from_le_bytes(node[40..42].try_into().unwrap()), 0);
        assert_eq!(u16::from_le_bytes(node[42..44].try_into().unwrap()), 64);
        assert_eq!(u16::from_le_bytes(node[44..46].try_into().unwrap()), 57);
        assert_eq!(longest_key, 24);
        assert_eq!(longest_val, 116);
        // The tree info at the end: flags 0x42, node size 4096, 4 keys, 1 node.
        let info = 4096 - 40;
        assert_eq!(
            u32::from_le_bytes(node[info..info + 4].try_into().unwrap()),
            0x42
        );
        assert_eq!(
            u64::from_le_bytes(node[info + 24..info + 32].try_into().unwrap()),
            4
        );
        assert_eq!(
            u64::from_le_bytes(node[info + 32..info + 40].try_into().unwrap()),
            1
        );
        // Keys ascend by the file-system key order.
        let ks = 56 + 64;
        let mut prev: Option<Vec<u8>> = None;
        for i in 0..4 {
            let (ko, kl) = (
                u16::from_le_bytes(node[56 + 8 * i..58 + 8 * i].try_into().unwrap()) as usize,
                u16::from_le_bytes(node[58 + 8 * i..60 + 8 * i].try_into().unwrap()) as usize,
            );
            let key = node[ks + ko..ks + ko + kl].to_vec();
            if let Some(p) = &prev {
                assert_eq!(crate::fs::compare_keys(p, &key), std::cmp::Ordering::Less);
            }
            prev = Some(key);
        }
    }
}
